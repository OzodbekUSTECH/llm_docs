from dishka import Provider, Scope, provide_all

from app.interactors.rules.create import CreateRuleInteractor
from app.interactors.rules.delete import DeleteRuleInteractor
from app.interactors.rules.update import UpdateRulePartiallyInteractor
from app.interactors.rules.get import GetAllRulesInteractor, GetRuleByIdInteractor
from app.interactors.rules.categories.create import CreateRuleCategoryInteractor
from app.interactors.rules.categories.delete import DeleteRuleCategoryInteractor
from app.interactors.rules.categories.update import UpdateRuleCategoryPartiallyInteractor
from app.interactors.rules.categories.get import GetAllRuleCategoriesInteractor, GetRuleCategoryByIdInteractor




class RulesInteractorProvider(Provider):

    scope = Scope.REQUEST

    interactors = provide_all(
        CreateRuleInteractor,
        DeleteRuleInteractor,
        UpdateRulePartiallyInteractor,
        GetAllRulesInteractor,
        GetRuleByIdInteractor,
        CreateRuleCategoryInteractor,
        DeleteRuleCategoryInteractor,
        UpdateRuleCategoryPartiallyInteractor,
        GetAllRuleCategoriesInteractor,
        GetRuleCategoryByIdInteractor
    )
