from typing import Literal, List
from pydantic import BaseModel, Field


class Syntax(BaseModel):
    """General morphology parameters for a conlang."""
    main_word_order: Literal["SOV", "SVO", "VSO", "VOS", "OSV", "OVS"] = Field(
        "VSO",
        description="The main word order of subject, object, and verb in the language.",
    )
    oblique_word_order = Literal["VOX", "VXO", "XOV", "XVO", "OVX", "OXV"] = Field(
        "VOX",
        description="The word order of object, oblique, and verb in the language."
    )
    adj_noun_word_order: Literal["AN", "NA"] = Field(
        "NA",
        description="The word order of adjectives and nouns in the language.",
    )
    posspron_noun_word_order: Literal["PossN", "NPoss"] = Field(
        "PossN",
        description="The word order of possessive pronouns and nouns in the language.",
    )
    num_noun_word_order: Literal["NumN", "NNum"] = Field(
        "NumN",
        description="The word order of numerals and nouns in the language.",
    )
    adposition_noun_word_order: Literal["PN", "NP"] = Field(
        "PN",
        description="The word order of adpositions (prepositions or postpositions) and nouns in the language.",
    )
    morphology_type: Literal["isolating", "agglutinative", "fusional"] = Field(
        "agglutinative",
        description="The type of morphology used in the language.",
    )
    alignment: Literal["nominative-accusative", "ergative-absolutive"] = Field(
        "nominative-accusative",
        description="The alignment system of the language, either nominative-accusative or ergative-absolutive.",
    )


class Relativization(BaseModel):
    """Relativization parameters for a conlang."""
    relativization_order: Literal["head-initial", "head-final"] = Field(
        "head-initial",
        description="The order of the relativized head and the relative clause in the language.",
    )
    relativization_marking: Literal["head-marking", "dependent-marking"] | None = Field(
        "head-marking",
        description="The type of relativization marking used in the language, either head-marking or dependent-marking.",
    )
    relativizer_position: Literal["prepositional", "postpositional"] | None = Field(
        "prepositional",
        description="The position of the relativizer in relation to the noun it modifies.",
    )
    relativizer_morpheme: Literal["affix", "word"] | None = Field(
        "affix",
        description="The type of relativizer used in the language, either an affix or a separate word.",
    )


class Voice(BaseModel):
    """Voice parameters for a conlang."""
    voice: List[Literal["active", "passive"]] = Field(
        "active",
        description="The voice used in the language, either active or passive.",
    )
    voice_marking_strategy: Literal["prefix", "suffix", "prepositional word", "postpositional word"] | None = Field(
        "prepositional word",
        description="The strategy used for voice marking, such as prefix, suffix, prepositional word, or postpositional word.",
    )


class Mood(BaseModel):
    """Mood parameters for a conlang."""
    mood: List[Literal["indicative", "subjunctive", "imperative", "conditional"]] | None = Field(
        None,
        description="List of moods used in the language, such as indicative, subjunctive, imperative, and conditional. If None, no mood marking is used.",
    )
    mood_marking_strategy: Literal["prefix", "suffix", "prepositional word", "postpositional word"] | None = Field(
        "suffix",
        description="The strategy used for mood marking, such as prefix, suffix, prepositional word, or postpositional word.",
    )


class Definiteness(BaseModel):
    """Definiteness parameters for a conlang."""
    definiteness: List[Literal["definite", "indefinite"]] | None = Field(
        None,
        description="List of definiteness markers used in the language, such as definite and indefinite. If None, no definiteness marking is used.",
    )
    definiteness_marking_strategy: Literal["prefix", "suffix", "prepositional word", "postpositional word"] | None = Field(
        "prepositional word",
        description="The strategy used for definiteness marking, such as prefix, suffix, prepositional word, or postpositional word.",
    )
    definiteness_agreement: List[Literal["gender", "number", "case"]] | None = Field(
        None,
        description="List of features that definiteness marker agrees with in the language, such as gender, number, and case. If None, no definiteness agreement is used.",
    )


class Case(BaseModel):
    """Case parameters for a conlang."""
    case_marking: List[Literal[
        "nominative", "accusative", "dative", "genitive",
        "ablative", "locative", "instrumental", "ergative", "absolutive"
    ]] | None = Field(
        None,
        description="List of cases used in the language, such as nominative, accusative, dative, etc. If None, no case marking is used.",
    )
    case_marking_strategy: Literal["prefix", "suffix", "prepositional word", "postpositional word"] | None = Field(
        "prepositional word",
        description="The strategy used for case marking, such as prefix, suffix, prepositional word, or postpositional word.",
    )
    oblique_case_marking: Literal[
        "nominative", "accusative", "dative", "genitive",
        "ablative", "locative", "instrumental", "ergative", "absolutive"
    ] | None = Field(
        None,
        description="The oblique case marking used in the language, such as genitive, dative, and accusative. If None, no oblique case marking is used.",
    )


class Infinitive(BaseModel):
    """Infinitive parameters for a conlang."""
    infinitive: Literal["infinitive"] | None = Field(
        "infinitive",
        description="The morpheme used to indicate the infinitive form of verbs in the language.",
    )
    infinitive_marking_strategy: Literal["prefix", "suffix", "prepositional word", "postpositional word"] = Field(
        "prepositional word",
        description="The position of the infinitive morpheme in relation to the verb it modifies.",
    )

class AdjectiveAgreement(BaseModel):
    """Adjective agreement parameters for a conlang."""
    adjective_agreement: List[Literal["gender", "number", "case", "definiteness"]] | None = Field(
        None,
        description="List of features that adjectives agree with in the language, such as gender, number, and case. If None, no adjective agreement is used.",
    )
    adjective_agreement_strategy: Literal["prefix", "suffix", "prepositional word", "postpositional word"] | None = Field(
        "suffix",
        description="The strategy used for adjective agreement, such as prefix, suffix, prepositional word, or postpositional word.",
    )


class Comparative(BaseModel):
    """Comparative parameters for a conlang."""
    comparative: List[Literal["comparative", "superlative", "equative"]] | None = Field(
        None,
        description="The type of comparative used in the language, either comparative or superlative. If None, no comparative marking is used.",
    )
    comparative_marking_strategy: Literal["prefix", "suffix", "prepositional word", "postpositional word"] | None = Field(
        "prepositional word",
        description="The strategy used for comparative marking, such as prefix, suffix, prepositional word, or postpositional word.",
    )


class TenseAspect(BaseModel):
    """Tense and aspect parameters for a conlang."""
    tense_aspect: List[Literal[
        "present", "past", "future", "imperfect", "perfective", "imperfective",
        "immediate past", "recent past", "remote past", "nonpast"
    ]] | None = Field(
        None,
        description=(
            "List of tenses and aspects used in the language, such as present, past, future, perfective, and imperfective. If None, no tense/aspect marking is used. "
            "Imperfect is a tense (or more accurately, combination of tense and aspect), while imperfective is strictly an aspect."
        )
    )
    tense_aspect_marking_strategy: Literal["prefix", "suffix", "prepositional word", "postpositional word"] | None = Field(
        "suffix",
        description="The strategy used for tense/aspect marking, such as prefix, suffix, prepositional word, or postpositional word.",
    )


class Person(BaseModel):
    """Person agreement markers."""
    person_agreement: List[Literal["first", "second", "third"]] | None = Field(
        None,
        description="List of person agreement markers used in the language, such as first, second, and third person. If None, no person agreement is used.",
    )
    person_marking_strategy: Literal["prefix", "suffix", "prepositional word", "postpositional word"] | None = Field(
        "suffix",
        description="The strategy used for person agreement marking, such as prefix, suffix, prepositional word, or postpositional word.",
    )
    verbal_number_agreement: List[Literal["singular", "plural", "dual", "paucal"]] | None = Field(
        None,
        description="List of verbal number agreement markers used in the language, such as singular, plural, dual, and paucal. If None, no verbal number agreement is used.",
    )
    verbal_number_marking_strategy: Literal["prefix", "suffix", "prepositional word", "postpositional word"] | None = Field(
        "suffix",
        description="The strategy used for verbal number agreement marking, such as prefix, suffix, prepositional word, or postpositional word.",
    )


class NominalNumber(BaseModel):
    """Nominal number parameters for a conlang."""
    nominal_number: List[Literal["singular", "plural", "dual", "paucal"]] | None = Field(
        None,
        description="List of nominal number markers used in the language, such as singular, plural, dual, and paucal. If None, no nominal number marking is used.",
    )
    nominal_number_marking_strategy: Literal["prefix", "suffix", "prepositional word", "postpositional word"] | None = Field(
        "suffix",
        description="The strategy used for nominal number marking, such as prefix, suffix, prepositional word, or postpositional word.",
    )


class Morphology(BaseModel, extra="allow"):
    """Morphology parameters for a conlang.
    Allow extra attributes to accommodate additional features, especially when
    handling a nested structure like `Relativization`.
    """
    pro_drop: Literal["pro-drop", "non-pro-drop"] = Field(
        "pro-drop",
        description="Whether the language is a pro-drop language or not.",
    )
    case: Case | None
    gender: List[Literal["masculine", "feminine", "neuter"]] | None = Field(
        None,
        description="List of genders used in the language, such as masculine, feminine, and neuter. If None, no gender marking is used.",
    )
    definiteness: Definiteness | None
    adjective_agreement: AdjectiveAgreement | None
    nominal_number: NominalNumber | None

    # verbal
    tense_aspect: TenseAspect | None
    person: Person | None
    voice: Voice | None
    mood: Mood | None
    relativization: Relativization
    infinitive: Infinitive | None
    negation: Literal["prefix", "suffix", "prepositional word", "postpositional word"] | None = Field(
        "prepositional word",
        description="The strategy used for negation marking, such as prefix, suffix, prepositional word, or postpositional word.",
    )

    # common
    inclusive_exclusive: bool | None = Field(
        None,
        description="Whether the language has inclusive and exclusive first-person plural distinctions.",
    )
    extras: List[str] | None = Field(
        None,
        description="Any additional inflectional features not covered by the above fields, such as clusivity, evidentiality, etc. If None, no extras are defined.",
    )


class Morphosyntax(BaseModel):
    syntax: Syntax
    morphology: Morphology
    

def sample_params_ainu():
    """Sample parameters based on Ainu morphology.
    Ainu is a language isolate with complex person marking on both verbs and nouns,
    polysynthetic tendencies, and extensive use of applicatives and voice changes.

    Key features:
    - SOV word order (typical of northern Japan region)
    - Complex person marking system with prefixes
    - No nominal case marking (but extensive verbal person marking)
    - Rich derivational morphology with many prefixes
    - Applicatives, causatives, and voice distinctions
    - Aspect marking through suffixes
    """
    return Morphosyntax(
        syntax=Syntax(
            main_word_order="SOV",
            oblique_word_order="XOV",
            adj_noun_word_order="AN",
            posspron_noun_word_order="PossN",
            num_noun_word_order="NumN",
            adposition_noun_word_order="NP",
            morphology_type="agglutinative",
            alignment="nominative-accusative"
        ),
        morphology=Morphology(
            pro_drop="pro-drop",
            case=None,
            gender=None,
            definiteness=None,
            adjective_agreement=None,
            comparative=None,
            tense_aspect=None,
            mood=None,
            voice=None,
            person=Person(
                person_agreement=["first", "second", "third"],
                person_marking_strategy="prefix",
                verbal_number_agreement=["singular", "plural"],
                verbal_number_marking_strategy="prefix"
            ),
            inclusive_exclusive=True,
            nominal_number=None,
            relativization=Relativization(
                relativization_order="head-final",
                relativization_marking=None,
                relativizer_position=None,
                relativizer_morpheme=None,
            ),
            negation="prepositional word",
            infinitive=None,
            extras=None
        )
    )


def sample_params_turkish():
    """Example parameters like Turkish."""
    return Morphosyntax(
        syntax=Syntax(
            main_word_order="SOV",
            oblique_word_order="XOV",
            adj_noun_word_order="AN",
            adposition_noun_word_order="NP",
            posspron_noun_word_order="PossN",
            num_noun_word_order="NumN",
            morphology_type="agglutinative",
            alignment="nominative-accusative"
        ),
        morphology=Morphology(
            pro_drop="pro-drop",
            case=Case(
                case_marking=["nominative", "accusative", "dative", "genitive", "ablative", "locative", "instrumental"],
                case_marking_strategy="suffix",
                oblique_case_marking="genitive"
            ),
            gender=None,
            definiteness=None,
            definiteness_marking_strategy=None,
            adjective_agreement=None,
            comparative=Comparative(
                comparative=["comparative", "superlative"],
                comparative_marking_strategy="prepositional word"  # e.g., daha, en
            ),
            tense_aspect=TenseAspect(
                tense_aspect=["present", "past", "future"],
                tense_aspect_marking_strategy="suffix"  # e.g., -iyor for present, -di for past, -ecek for future
            ),
            mood=Mood(
                mood=["indicative", "imperative", "conditional"],
                mood_marking_strategy="suffix"
            ),
            voice=Voice(
                voice=["active", "passive"],
                voice_marking_strategy="suffix"
            ),
            person=Person(
                person_agreement=["first", "second", "third"],
                person_marking_strategy="suffix",
                verbal_number_agreement=["singular", "plural"],
                verbal_number_marking_strategy="suffix",
            ),
            inclusive_exclusive=None,
            nominal_number=NominalNumber(
                nominal_number=["singular", "plural"],
                nominal_number_marking_strategy="suffix",
            ),
            relativization=Relativization(
                relativization_order="head-final",
                relativization_marking="dependent-marking",
                relativizer_position="postpositional",
                relativizer_morpheme="affix", # dik-
            ),
            negation="suffix",
            infinitive=Infinitive(
                infinitive="infinitive",
                infinitive_position="suffix"  # -mek/-mak
            ),
        )
    )


def sample_params_french():
    """Sample parameters like French."""
    return Morphosyntax(
        syntax=Syntax(
            main_word_order="SVO",
            oblique_word_order="VOX",
            adj_noun_word_order="NA",
            posspron_noun_word_order="PossN",
            num_noun_word_order="NumN",
            adposition_noun_word_order="PN",
            morphology_type="fusional",
            alignment="nominative-accusative"
        ),
        morphology=Morphology(
            pro_drop="non-pro-drop",
            case=None,
            gender=["masculine", "feminine"],
            definiteness=Definiteness(
                definiteness=["definite", "indefinite"],
                definiteness_marking_strategy="prepositional word",
                definiteness_agreement=["gender", "number"]
            ),
            definiteness_marking_strategy="prepositional word",
            adjective_agreement=AdjectiveAgreement(
                adjective_agreement=["gender", "number"],
                adjective_agreement_strategy="suffix"
            ),
            comparative=Comparative(
                comparative=["comparative", "superlative", "equative"],
                comparative_marking_strategy="prepositional word"  # e.g., plus, aussi, moins
            ),
            tense_aspect=TenseAspect(
                tense_aspect=["present", "past", "future", "imperfect"],
                tense_aspect_marking_strategy="suffix"
            ),
            mood=Mood(
                mood=["indicative", "subjunctive", "imperative", "conditional"],
                mood_marking_strategy="suffix"
            ),
            person=Person(
                person_agreement=["first", "second", "third"],
                person_marking_strategy="suffix",
                verbal_number_agreement=["singular", "plural"],
                verbal_number_marking_strategy="suffix",
            ),
            voice=Voice(
                voice=["active", "passive"],
                voice_marking_strategy="suffix"
            ),
            inclusive_exclusive=None,
            nominal_number=NominalNumber(
                nominal_number=["singular", "plural"],
                nominal_number_marking_strategy="suffix",
            ),
            relativization=Relativization(
                relativization_order="head-initial",
                relativization_marking="head-marking",
                relativizer_position="postpositional",
                relativizer_morpheme="word", # que
            ),
            negation="postpositional word", # pas
            infinitive=Infinitive(
                infinitive="infinitive",
                infinitive_position="suffix"  # -er
            ),
        )
    )


def sample_params_arabic():
    """Sample feature set like Arabic."""
    return Morphosyntax(
        syntax=Syntax(
            main_word_order="VSO",
            oblique_word_order="VOX",
            adj_noun_word_order="NA",
            posspron_noun_word_order="NPoss",
            num_noun_word_order="NumN",
            adposition_noun_word_order="PN",
            morphology_type="fusional",
            alignment="nominative-accusative"
        ),
        morphology=Morphology(
            pro_drop="non-pro-drop",
            case=Case(
                case_marking=["nominative", "accusative", "genitive"],
                case_marking_strategy="suffix",
                oblique_case_marking="genitive"  # genitive is the oblique case
            ),
            gender=["masculine", "feminine"],
            definiteness=Definiteness(
                definiteness=["definite"],
                definiteness_marking_strategy="prefix",
                definiteness_agreement=None,
            ),
            adjective_agreement=AdjectiveAgreement(
                adjective_agreement=["gender", "number", "case", "definiteness"],
                adjective_agreement_strategy="suffix"
            ),
            comparative=Comparative(
                comparative=["comparative", "superlative"],
                comparative_marking_strategy="suffix"  # tentatively; it's more complex in reality
            ),
            tense_aspect=TenseAspect(
                tense_aspect=["present", "past", "future"],
                tense_aspect_marking_strategy="suffix"  # e.g., -a for past (it's more fusional but just for simplicity)
            ),
            mood=Mood(
                mood=["indicative", "subjunctive", "imperative"], # there's also jussive but omitted for simplicity
                mood_marking_strategy="suffix"
            ),
            voice=Voice(
                voice=["active", "passive"],
                voice_marking_strategy="suffix"
            ),
            person=Person(
                person_agreement=["first", "second", "third"],
                person_marking_strategy="suffix",
                verbal_number_agreement=["singular", "dual", "plural"],
                verbal_number_marking_strategy="suffix",
            ),
            inclusive_exclusive=None,
            nominal_number=NominalNumber(
                nominal_number=["singular", "dual", "plural"],
                nominal_number_marking_strategy="suffix",
            ),
            relativization=Relativization(
                relativization_order="head-initial",
                relativization_marking="head-marking",
                relativizer_position="postpositional",
                relativizer_morpheme="word", # alla
            ),
            negation="prepositional word",
            infinitive=None,
        )
    )


def sample_params_welsh():
    """Sample feature set like (Colloquial) Welsh.
    We are not considering the consonant mutation system here.

    An example of a Welsh sentence:
    Ni  roddais         i   ddim    llyfr da    i   dad     Eleri   ddoe.
    NEG give.1SG.PST    1SG NEG     book  good  to  father  Eleri   yesterday.
    """
    return Morphosyntax(
        syntax=Syntax(
            main_word_order="VSO",
            oblique_word_order="VOX",
            adj_noun_word_order="NA",
            posspron_noun_word_order="NPoss",
            num_noun_word_order="NumN",
            adposition_noun_word_order="PN",
            morphology_type="fusional",
            alignment="nominative-accusative"
        ),
        morphology=Morphology(
            pro_drop="non-pro-drop",
            case=None,
            gender=["masculine", "feminine"],
            definiteness=Definiteness(
                definiteness=["definite"],
                definiteness_marking_strategy="prepositional word",
                definiteness_agreement=None,
            ),
            adjective_agreement=None,
            comparative=Comparative(
                comparative=["comparative", "superlative"],
                comparative_marking_strategy="suffix" # -ach, -a
            ),
            tense_aspect=TenseAspect(
                tense_aspect=["present", "past", "future"],
                tense_aspect_marking_strategy="suffix"
            ),
            mood=Mood(
                mood=["indicative", "subjunctive", "conditional", "imperative"],
                mood_marking_strategy="suffix"
            ),
            voice=None,
            person=Person(
                person_agreement=["first", "second", "third"],
                person_marking_strategy="suffix",
                verbal_number_agreement=["singular", "plural"],
                verbal_number_marking_strategy="suffix",
            ),
            inclusive_exclusive=None,
            nominal_number=NominalNumber(
                nominal_number=["singular", "plural"],
                nominal_number_marking_strategy="suffix",
            ),
            relativization=Relativization(
                relativization_order="head-initial",
                relativization_marking="head-marking",
                relativizer_position="postpositional",
                relativizer_morpheme="word", # a or y
            ),
            negation="prepositional word",
            infinitive=None,
        )
    )


def sample_params_vietnamese():
    """Sample feature set like Vietnamese.
    Vietnamese is an isolating language, so it has no inflectional morphology.
    """
    return Morphosyntax(
        syntax=Syntax(
            main_word_order="SVO",
            oblique_word_order="VOX",
            adj_noun_word_order="NA",
            posspron_noun_word_order="NPoss",
            num_noun_word_order="NumN",
            adposition_noun_word_order="PN",
            morphology_type="isolating",
            alignment="nominative-accusative"
        ),
        morphology=Morphology(
            pro_drop="pro-drop",
            case=None,
            gender=None,
            definiteness=None,
            definiteness_marking_strategy=None,
            adjective_agreement=None,
            comparative=Comparative(
                comparative=["comparative", "superlative", "equative"],
                comparative_marking_strategy="postpositional word"  # e.g., hơn, nhất, bằng
            ),
            tense_aspect=None,
            mood=None,
            voice=None,
            person=None,
            inclusive_exclusive=None,
            nominal_number=None,
            relativization=Relativization(
                relativization_order="head-initial",
                relativization_marking="head-marking",
                relativizer_position="postpositional",
                relativizer_morpheme="word",  # mà
            ),
            negation="prepositional word", # không
            infinitive=None,
        )
    )


def sample_params_mizo():
    """Sample feature set like Mizo.
    Wikipedia says Mizo is OSV, but it seems it's actually SOV with
    ergative-absolutive alignment. It sometimes seems like it has a OSV
    word order because the subject particle (clitic?) always immediately
    precedes the verb. However, a non-pronominal transitive subject is
    expressed as an ergative noun phrase, which comes before the absolutive
    object.
    For this conlang project, we experimentally change the word order to OSV
    for convenience. 

    Reference: http://sealang.net/sala/archives/pdf8/chhangte1989grammar.pdf
    """
    return Morphosyntax(
        syntax=Syntax(
            main_word_order="OSV",
            oblique_word_order="XOV",
            adj_noun_word_order="NA",
            posspron_noun_word_order="PossN",
            num_noun_word_order="NNum",
            adposition_noun_word_order="NP",
            morphology_type="isolating",
            alignment="ergative-absolutive"  # Mizo is ergative-absolutive
        ),
        morphology=Morphology(
            pro_drop="pro-drop",
            case=Case(
                case_marking=["ergative", "absolutive", "genitive", "instrumental"],
                case_marking_strategy="postpositional word",
                oblique_case_marking="absolutive"
            ),
            gender=None,
            definiteness=None,
            definiteness_marking_strategy="postpositional word",
            adjective_agreement=None,
            comparative=Comparative(
                comparative=["comparative", "superlative"],
                comparative_marking_strategy="postpositional word"  # e.g., zook3, ber
            ),
            tense_aspect=None,
            mood=None,
            voice=None,
            person=Person(
                person_agreement=["first", "second", "third"],
                person_marking_strategy="prepositional word",
                verbal_number_agreement=["singular", "plural"],
                verbal_number_marking_strategy="prepositional word",
            ),
            inclusive_exclusive=None,
            nominal_number=None,
            relativization=Relativization(
                relativization_order="head-initial",
                relativization_marking="dependent-marking",
                relativizer_position="postpositional",
                relativizer_morpheme="affix",  # -a
            ),
            negation="postpositional word", # lou
            infinitive=None,
        )
    )


def sample_params_fijian():
    """Sample feature set like Fijian.
    Fijian is a SVO language with ergative-absolutive alignment.
    It has no inflectional morphology, so it is isolating.
    Also, for `oblique_word_order`, WALS reports that Mizo
    has no dominat order; for convenience, we borrow Malagasy's config (VOX)
    which is also a VOS language.

    Reference: http://www.aa.tufs.ac.jp/elib/ltext/fji/pdf/a.pdf
    """
    return Morphosyntax(
        syntax=Syntax(
            main_word_order="VOS",
            oblique_word_order="VOX",
            adj_noun_word_order="NA",
            posspron_noun_word_order="NPoss",
            num_noun_word_order="NumN",
            adposition_noun_word_order="PN",
            morphology_type="isolating",
            alignment="nominative-accusative"
        ),
        morphology=Morphology(
            pro_drop="non-pro-drop",
            case=None,
            gender=None,
            definiteness=None,
            definiteness_marking_strategy=None,
            adjective_agreement=None,
            comparative=Comparative(
                comparative=["comparative", "superlative"],
                comparative_marking_strategy="postpositional word"  # e.g., ca'e
            ),
            tense_aspect=None,
            mood=None,
            voice=None,
            person=Person(
                person_agreement=["first", "second", "third"],
                person_marking_strategy="prepositional word",  # e.g., o, e
                verbal_number_agreement=["singular", "dual", "paucal", "plural"],
                verbal_number_marking_strategy="prepositional word",  # e.g., o,
            ),
            inclusive_exclusive=None,
            nominal_number=None,
            relativization=Relativization(
                relativization_order="head-initial",
                relativization_marking=None,
                relativizer_position=None,
                relativizer_morpheme=None, # head-initial but no morphological REL marking
            ),
            negation="prepositional word", # sega
            infinitive=None,
        )
    )


def sample_params_hixkaryana():
    """Sample feature set like Hixkaryana.
    Hixkaryana is a OVS language with PN word order, with inclusive-exclusive distinction, PossN word order,
    NumN, AN, no passive,
    Relative clause is formed by a head-final, dependent-marking, post-positional affix.
    Tense is quite complicated...:
    - nonpast
    - nonpast uncertain
    - immediate past
    - recent past completive
    - recent past continuative
    - distant past completive
    - distant past continuative

    References:
    - https://static1.squarespace.com/static/586960e5197aea52834230a2/t/58c469a0e4fcb5898b20f55b/1489267124489/Kalin-Hixkaryana-WCCFL
    - http://136.175.10.10/ebook/pdf/Hixkaryana_and_Linguistic_Typology.pdf
    """
    return Morphosyntax(
        syntax=Syntax(
            main_word_order="OVS",
            oblique_word_order="OVX",
            adj_noun_word_order="AN",
            posspron_noun_word_order="PossN",
            num_noun_word_order="NumN",
            adposition_noun_word_order="PN",
            morphology_type="fusional",
            alignment="nominative-accusative"
        ),
        morphology=Morphology(
            pro_drop="pro-drop",
            case=None,
            gender=None,
            definiteness=None,
            definiteness_marking_strategy=None,
            adjective_agreement=None,
            comparative=Comparative(
                comparative=["comparative", "superlative", "equative"],
                comparative_marking_strategy="postpositional word"  # e.g., nyhe (more), rmahaxa (very much; most), rye (same)
            ),
            tense_aspect=TenseAspect(
                tense_aspect=[
                "nonpast", "immediate past", # nonpast uncertain, immediate past are omitted for simplicity.
                "recent past", # recent past continuative and recent past completive are merged to recent past for simplicity.
                "remote past" # distant past continuative and distant past completive are merged to distant past for simplicity.
            ],
                tense_aspect_marking_strategy="suffix"
            ),
            mood=None,
            voice=None,
            person=Person(
                person_agreement=["first", "second", "third"],
                person_marking_strategy="suffix" ,
                verbal_number_agreement=["singular", "plural"],
                verbal_number_marking_strategy="suffix"
            ),
            inclusive_exclusive=True,
            nominal_number=NominalNumber(
                nominal_number=["singular", "plural"],
                nominal_number_marking_strategy="suffix",
            ),
            relativization=Relativization(
                relativization_order="head-final",
                relativization_marking="dependent-marking",
                relativizer_position="postpositional",
                relativizer_morpheme="affix",
            ),
            negation="suffix", # it's actually more complicated; See Derbyshire 1985
            infinitive=None,
        )
    )


def sample_params_hard():
    """Sample feature set like a hard language.
    A language with typologically unusual morphosyntax.

    main word order: OSV
    oblique-object-verb order: OXV
    adj-noun oder: NA
    posspron-noun order: NPoss
    num-noun order: NNum (NumN is actually rarer in terms of the number of languages,
        but NNum centers around Africa and South East Asia, where many languages are underrepresented.)
    adposition-noun order: NP (since we go for head-initial in adj-noun, we go for head-final here for weirdness)
    morphology type: fusional
    alignment: ergative-absolutive
    pro-drop: pro-drop (maybe we should do away with pro-drop.)
    case: ergative, absolutive, genitive, dative, locative, instrumental
        strategy: prefix
    gender: None (for now)
    definiteness: definite, indefinite
        strategy: suffix
    comparative: comparative, superlative, equative
        strategy: prefix
    tense-aspect: present, future, recent past, remote past
        strategy: prefix
    mood: indicative, subjunctive, imperative, conditional
        strategy: prefix
    voice: active, passive
        strategy: prefix
    person: first, second, third
        strategy: suffix
        verbal number: singular, plural, dual
        strategy: prefix
    inclusive-exclusive: yes
    nominal number: singular, plural, dual
        strategy: prefix
    relativization: head-final, head-marking, postpositional affix
    negation: suffix
    infinitive: prefix
    """
    return Morphosyntax(
        syntax=Syntax(
            main_word_order="OSV",
            oblique_word_order="OXV",
            adj_noun_word_order="NA",
            posspron_noun_word_order="NPoss",
            num_noun_word_order="NNum",
            adposition_noun_word_order="NP",
            morphology_type="fusional",
            alignment="ergative-absolutive"
        ),
        morphology=Morphology(
            pro_drop="non-pro-drop",
            case=Case(
                case_marking=["ergative", "absolutive", "genitive", "dative", "locative", "instrumental"],
                case_marking_strategy="prefix",
                oblique_case_marking="instrumental"
            ),
            gender=None,
            definiteness=Definiteness(
                definiteness=["definite", "indefinite"],
                definiteness_marking_strategy="suffix",
                definiteness_agreement=None,
            ),
            adjective_agreement=AdjectiveAgreement(
                adjective_agreement=["number", "case", "definiteness"],
                adjective_agreement_strategy="prefix"
            ),
            comparative=Comparative(
                comparative=["comparative", "superlative", "equative"],
                comparative_marking_strategy="prefix"
            ),
            tense_aspect=TenseAspect(
                tense_aspect=["present", "future", "recent past", "remote past"],
                tense_aspect_marking_strategy="prefix"
            ),
            mood=Mood(
                mood=["indicative", "subjunctive", "imperative", "conditional"],
                mood_marking_strategy="prefix"
            ),
            voice=Voice(
                voice=["active", "passive"],
                voice_marking_strategy="prefix"
            ),
            person=Person(
                person_agreement=["first", "second", "third"],
                person_marking_strategy="suffix",
                verbal_number_agreement=["singular", "plural", "dual"],
                verbal_number_marking_strategy="prefix",
            ),
            inclusive_exclusive=True,
            nominal_number=NominalNumber(
                nominal_number=["singular", "plural", "dual"],
                nominal_number_marking_strategy="prefix",
            ),
            relativization=Relativization(
                relativization_order="head-final",
                relativization_marking="head-marking",
                relativizer_position="postpositional",
                relativizer_morpheme="affix",
            ),
            negation="suffix",
            infinitive=Infinitive(
                infinitive="infinitive",
                infinitive_position="prefix"
            ),
        )
    )


LANGUAGE_TO_PARAMS = {
    "ainu": sample_params_ainu,
    "turkish": sample_params_turkish,
    "french": sample_params_french,
    "arabic": sample_params_arabic,
    "welsh": sample_params_welsh,
    "vietnamese": sample_params_vietnamese,
    "mizo": sample_params_mizo,
    "fijian": sample_params_fijian,
    "hixkaryana": sample_params_hixkaryana,
    "hard": sample_params_hard,
}
